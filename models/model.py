import os, gc, time, csv
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
from tqdm import tqdm

from guided_diffusion.dist_util import dev
from networks.ddpm import prepare_concat_features
from networks.seg_head import pixel_classifier

from utils.metrics import AverageMeter, RunningScore, multi_acc, save_scores_to_csv, batch_miou
from utils.data_util import Visualiser
from utils.train_util import get_optimizer, get_lr_scheduler, warmup_scheduler
from utils.utils import (
    get_dataloader, get_model, predict_labels, visualize_accumulated_queries, make_noise,
    write_accumulated_query, count_parameters, get_img_basename,
    load_candidates, save_last_queries, load_last_queries, round_status, last_finished_round
)
from query import QuerySelector

class Model:
    def __init__(self, args):
        self.args = args
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss = float("inf")
        self.patience = 50
        self.target_accuracy = 0.95
        self.break_count = 0
        
        self.best_miou = -1.0
        self.best_epoch = -1
        
        if args.use_wandb:
            import wandb
            self.wandb = wandb
        else:
            class _NoWandb:
                def __getattr__(self, _):
                    return lambda *__, **___: None
            self.wandb = _NoWandb()
        
        self.dataloader_train = get_dataloader(args, batch_size=args.batch_size, shuffle=False, val=False, query=False)
        shared_subset = (
            self.dataloader_train.dataset.list_inputs,
            self.dataloader_train.dataset.list_labels,
            self.dataloader_train.dataset.queries,
            self.dataloader_train.dataset.global_queries,
            self.dataloader_train.dataset.local_candidates
        )
        self.dataloader_query = get_dataloader(args, batch_size=1, shuffle=False, val=False, query=True, shared_subset=shared_subset)
        self.dataloader_eval = get_dataloader(args, batch_size=args.batch_size, shuffle=False, val=True, query=False)
        self.query_selector = QuerySelector(args, self.dataloader_query)
        self.vis = Visualiser(args, args.dataset_name)
        self.running_loss, self.running_score = AverageMeter(), RunningScore(n_classes=args.n_classes, ignore_index=self.ignore_index, dataset_name=args.dataset_name)
    
    def _refresh_query_dataloader(self):
        shared_subset = (
            self.dataloader_train.dataset.list_inputs,
            self.dataloader_train.dataset.list_labels,
            self.dataloader_train.dataset.queries,
            self.dataloader_train.dataset.global_queries,
            self.dataloader_train.dataset.local_candidates
        )
        self.dataloader_query = get_dataloader(
            deepcopy(self.args), batch_size=1, shuffle=False, val=False, query=True,
            shared_subset=shared_subset
        )
        self.query_selector = QuerySelector(self.args, self.dataloader_query)

    def __call__(self):
        self.wandb.init(
            project="two-stage-eDALD",
            name=f"{self.args.dataset_name}",
            config=vars(self.args),
            resume="allow" if self.args.resume else None
        )

        # ===== 1) Fully-Supervised Mode =====
        if self.final_budget_factor == 0:
            dir_ckpt = f"{self.dir_checkpoints}/fully_sup"
            os.makedirs(dir_ckpt, exist_ok=True)
            self.log_train = f"{dir_ckpt}/log_train.md"
            ckpt_paths = [os.path.join(dir_ckpt, f"model_{i}.pth") for i in range(self.args.model_num)]
            pretrained = [os.path.exists(p) for p in ckpt_paths]
            self.args.start_model_num = sum(pretrained)

            if all(pretrained):
                self._eval()
            else:
                if self.args.train_mode == "early_stop":
                    self._train()
                    self._eval(train_mode="early_stop")
            return

        # ===== 2) Active Learning Mode =====
        self.args.start_model_num = 0
        n_stages = self.n_stages
        start_round = 0
        skip_training = False
        queries = None
        local_cands = None

        print("n_stages:", n_stages)

        # ---- Resume Logic ----
        if self.args.resume:
            queries, last_round_saved = load_last_queries(self.dir_checkpoints, self.args.seed)

            if last_round_saved >= 0:
                latest_ckpt_round = last_finished_round(self.dir_checkpoints)
                status = round_status(self.dir_checkpoints, latest_ckpt_round, last_round_saved)

                if status == "not_trained":
                    start_round = last_round_saved
                    skip_training = False
                elif status == "trained_only":
                    start_round = last_round_saved
                    skip_training = True
                elif status == "evaluated":
                    start_round = last_round_saved
                    skip_training = True

                print(f"[RESUME] Round {last_round_saved} status = {status} -> start_round = {start_round}")
                if queries is not None:
                    total_pix = sum(q.sum() for q in queries)
                    print(f"Cumulative labeled pixels so far: {total_pix}")

                list_fp = os.path.join(self.dir_checkpoints, f"list_inputs_seed{self.args.seed}.txt")
                with open(list_fp) as f:
                    fixed_inputs = [ln.strip() for ln in f]
                fixed_labels = [p.replace(f".{self.img_ext}", ".npy") for p in fixed_inputs]

                local_cands = load_candidates(self.dir_checkpoints, 0, self.args.seed)

                queries_val = queries if queries is not None else self.dataloader_train.dataset.queries
                shared_subset = (
                    fixed_inputs,
                    fixed_labels,
                    queries_val,
                    queries_val,  # global_queries (same as queries for resume)
                    local_cands
                )
                self.dataloader_train = get_dataloader(
                    deepcopy(self.args), batch_size=self.args.batch_size,
                    shuffle=False, val=False, query=False,
                    shared_subset=shared_subset
                )
                self._refresh_query_dataloader()

            print(f"Active-Learning will start from round {start_round}")

        # ---- Active Learning Loop ----
        for nth_query in range(start_round, n_stages):
            print("\n" + "=" * 50)
            print(f"Starting {nth_query}/{n_stages-1} stage")
            print("=" * 50 + "\n")
            self.wandb.log({"active_round": nth_query})

            dir_ckpt = f"{self.dir_checkpoints}/{nth_query}_query"
            os.makedirs(dir_ckpt, exist_ok=True)
            self.log_train = f"{dir_ckpt}/log_train.md"
            self.nth_query = nth_query

            # === 2.1) Resume with Skip-Training ===
            if skip_training and nth_query == start_round:
                ckpt_path = os.path.join(dir_ckpt, "model_0.pth")
                ckpt = torch.load(ckpt_path)
                concat_clf = nn.DataParallel(
                    pixel_classifier(self.args, self.args.n_classes, dim=ckpt["sumC"])
                )
                concat_clf.load_state_dict(ckpt["model_state_dict"])
                concat_clf = concat_clf.module.to(self.device).eval()

                model_fq = {"concat": concat_clf, "blocks": [concat_clf]}
                if status == "trained_only":
                    self._eval(classifiers=[concat_clf], train_mode="early_stop")
                elif status == "evaluated":
                    print(f"[RESUME] skipping train & eval, re-running query for round {nth_query+1}")

                self._refresh_query_dataloader()
                result = self.query_selector(nth_query, model_fq)
                dq, dgq = result
                self.dataloader_train.dataset.label_queries(dq, dgq)

                save_last_queries(self.dataloader_train.dataset, self.dir_checkpoints, nth_query + 1, self.args.seed)
                skip_training = False
                continue

            # === 2.2) Training ===
            concat_dict = self._train()
            concat_clf = concat_dict["classifiers"][0]

            model_fq = {"concat": concat_clf, "blocks": [concat_clf]}

            self._eval(train_mode="early_stop")

            # === 2.3) Query Selection ===
            self._refresh_query_dataloader()
            result = self.query_selector(nth_query, model_fq)
            dq, dgq = result
            self.dataloader_train.dataset.label_queries(dq, dgq)

            save_last_queries(self.dataloader_train.dataset, self.dir_checkpoints, nth_query + 1, self.args.seed)

            if nth_query == n_stages - 1:
                break

        print("Active learning completed for all stages!")
        self.wandb.finish()

    def _train(self, round_iteration=0):
        args = self.args
        self.lr_history = []
        self.loss_history = []
        print(f"Checkpoint Directory:\n{self.dir_checkpoints}\n")
        base_dir = f"{self.dir_checkpoints}/fully_sup" if self.final_budget_factor == 0 else f"{self.dir_checkpoints}/{self.nth_query}_query"
        
        print(f"\nTraining with Early Stopping...\n")
        train_loader = self.dataloader_train
        print(f"Number of iteration per epoch: {len(train_loader)}")
        write_accumulated_query(train_loader, base_dir, args, self.nth_query)
        visualize_accumulated_queries(train_loader, f"{base_dir}/accumulated_queries", self.args.dataset_name)

        feature_extractor = get_model(self.args).to(self.device)
        feature_extractor.model.train()
        
        noise = make_noise(self.args)
        
        total_start_time = time.time()
        for MODEL_NUMBER in range(args.start_model_num, args.model_num, 1):
            gc.collect()
            torch.cuda.empty_cache()

            classifier = pixel_classifier(args, n_classes=args.n_classes, dim=args.dim)
            classifier.init_weights()
            classifier = nn.DataParallel(classifier).cuda()
            classifier.train()
            seg_head_params = count_parameters(classifier)
            best_loss = float('inf')
            iteration = 0

            warmup_epochs = args.warmup_epochs
            initial_lr = args.optimizer_params["initial_lr"]

            print('-------------- Trainable Parameters -----------------')
            print(f"| Segmentation Head Parameters    : {seg_head_params:,}")
            print('------------------------------------------------------')
            criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

            optimizer = get_optimizer(self.args, feature_extractor, classifier)
            scheduler = get_lr_scheduler(self.args, optimizer)
                        
            best_loss = float('inf')
            iteration = 0
            break_count = 0
            stop_sign = 0
            warmup_epochs = args.warmup_epochs
            initial_lr = args.optimizer_params["initial_lr"]
            learning_rates = []
            loss_history = []

            for epoch in range(self.n_epochs):
                warmup_scheduler(optimizer, epoch, warmup_epochs, initial_lr)
                self.wandb.log({"epoch": epoch, "active_round": self.nth_query})
                for i, dict_data in enumerate(train_loader):
                    queries: torch.Tensor = dict_data["queries"]
                    if queries.sum().item() == 0:
                        print("Skip this batch because all label is masked.")
                        continue
                    start_time = time.time()
                    batch_img, batch_label = dict_data['x'].to(dev()), dict_data['y'].to(dev())

                    if self.final_budget_factor != 0:
                        mask = dict_data['queries'].to(self.device)
                        batch_label = batch_label.clone()
                        batch_label.flatten()[~mask.flatten()] = self.ignore_index
                    
                    processed_images, processed_labels = prepare_concat_features(
                        batch_img, batch_label, feature_extractor, args, noise)
                    
                    X_batch = processed_images.to(dev()).float()
                    y_batch = processed_labels.to(dev()).type(torch.long)

                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    loss_history.append(loss.item())
                    current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                    learning_rates.append(current_lrs)   
                                     
                    pred = y_pred.argmax(dim=1)
                    
                    self.running_score.update([y_batch.cpu().numpy()], [pred.cpu().numpy()])
                    scores_dict, class_names, class_ious, class_accs = self.running_score.get_scores(return_classwise=True)
                    miou = scores_dict["Mean IoU"]
                    
                    self.wandb.log({
                        "train/loss": loss.item(),
                        "train/acc": acc.item(),
                        "train/mIoU": miou,
                        "learning_rate/classifier": current_lrs[0],
                        "iteration": round_iteration,
                        "epoch": epoch,
                        "active_round": self.nth_query
                    })

                    batch_time = time.time() - start_time
                    print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item():.4f}, Pixel Acc: {acc:.4f}, batch_time: {batch_time:.3f}')
                    round_iteration += 1
                    iteration += 1

                    del X_batch, y_batch, processed_images, processed_labels
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if i % 200 == 0:
                        with torch.no_grad():
                            prob = F.softmax(y_pred, dim=1)
                            pred = y_pred.argmax(dim=1)

                            if self.args.segmentation_head == "mlp":
                                N = batch_img.shape[0]
                                H, W = dict_data['y'].shape[1:]
                                pred = pred.view(N, H, W)

                            ent, lc, ms = [self._query(prob, uc)[0].cpu() for uc in ["entropy", "least_confidence", "margin_sampling"]]

                            dict_tensors = {
                                'input': batch_img[0].cpu(),
                                'target': dict_data['y'][0].cpu(),
                                'pred': pred[0].detach().cpu(),
                                'confidence': lc,
                                'margin': ms,
                                'entropy': ent
                            }
                            vis_save_path = f"{base_dir}/train/{epoch}_{i}_train.png"

                            self.vis(dict_tensors, fp=vis_save_path)
                    
                    if epoch > 3:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 50:
                            if acc > self.target_accuracy:
                                stop_sign = 1
                                print(f"*************** Early Stopping Triggered: Accuracy {acc:.2f}% ***************")
                                break
                            else:
                                print(f"Accuracy {acc:.2f}% is below target. Resetting break count.")
                                break_count = 0

                if epoch >= warmup_epochs:
                    scheduler.step()
                print(f'Epoch {epoch} completed. Pixel Accuracy: {acc:.4f}, Loss: {loss.item():.4f}')

                if stop_sign == 1:
                    break
            print(f'Finished training model {MODEL_NUMBER} at epoch: %d, Learning Rate: %s' % (epoch + 1, scheduler.get_last_lr()))   
            
            classifier_path = os.path.join(base_dir, f"model_{MODEL_NUMBER}.pth")
            torch.save({'model_state_dict': classifier.state_dict(), 'sumC': args.dim}, classifier_path)
            print('Saved MLP model.')
        
        print('MLP Training complete.')
        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        print('Duration (Total training time): %.2f seconds' % total_training_time) 
        
        return {"classifiers": [classifier],
                "noise": noise}
    
    @torch.no_grad()
    def _eval(self, classifiers=None, feature_extractor=None, noise=None, epoch=None, train_mode="early_stop"):
        args = self.args
        print("\nStarting Evaluation...\n")

        if feature_extractor is None:
            feature_extractor = get_model(self.args).to(self.device)
            feature_extractor.model.eval()
        noise = make_noise(self.args)

        base_dir = f"{self.dir_checkpoints}/fully_sup" if self.final_budget_factor == 0 else f"{self.dir_checkpoints}/{self.nth_query}_query"

        if classifiers is None:
            classifiers = []
            for i in range(self.args.model_num):
                model_path = os.path.join(base_dir, f"model_{i}.pth")
                checkpoint = torch.load(model_path)
                state_dict = checkpoint["model_state_dict"]
                classifier = nn.DataParallel(pixel_classifier(args, self.args.n_classes, dim=checkpoint["sumC"]))
                classifier.load_state_dict(state_dict)
                classifier = classifier.module.to(self.device).eval()
                classifiers.append(classifier)

        uncertainty_scores, variance_scores = [], []
        per_img_mious, global_idx = [], 0

        pbar = tqdm(self.dataloader_eval, desc="Evaluating", dynamic_ncols=True)

        for i, batch in enumerate(pbar):
            img = batch["x"].to(self.device)
            label = batch["y"].to(self.device)
            feats_cat, _ = prepare_concat_features(img, label, feature_extractor, args, noise)
            pred, u_score, prob = predict_labels(classifiers, feats_cat, label.shape, args)

            label_np = label.cpu().numpy()
            pred_np = pred.cpu().numpy()
            self.running_score.update(list(label_np), list(pred_np))

            uncertainty_scores.append(u_score.item())

            miou_batch = batch_miou(pred.cpu(), label.cpu(), args.n_classes)
            B = miou_batch.size(0)
            for k in range(B):
                img_path = self.dataloader_eval.dataset.list_inputs[global_idx + k]
                img_name = get_img_basename(img_path)
                per_img_mious.append((img_name, float(miou_batch[k])))
            global_idx += B

            var_map = torch.var(prob, dim=1)
            variance_scores.append(var_map.mean().item())

            scores, class_names, class_ious, class_accs = self.running_score.get_scores(return_classwise=True)
            miou = scores["Mean IoU"]
            pixel_acc = scores["Pixel Acc"]
            mean_class_acc = scores["Mean Class Acc"]

            self.wandb.log({
                "val/mIoU": miou,
                "val/pixel_acc": pixel_acc,
                "val/mean_class_acc": mean_class_acc,
                "epoch": -1 if epoch is None else epoch,
                "active_round": self.nth_query
            })
            pbar.set_postfix(mIoU=f"{miou:.4f}", Acc=f"{pixel_acc:.4f}", Var=f"{variance_scores[-1]:.5f}")

            image_path = self.dataloader_eval.dataset.list_inputs[i]
            img_base = get_img_basename(image_path)
            vis_path = os.path.join(base_dir, f"eval/{img_base}.png")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            ent, lc, ms = [self._query(prob, uc)[0].cpu() for uc in ("entropy", "least_confidence", "margin_sampling")]
            self.vis({
                "input": img[0].cpu(),
                "target": label[0].cpu(),
                "pred": pred[0].detach().cpu(),
                "confidence": lc,
                "margin": ms,
                "entropy": ent,
            }, fp=vis_path)

        per_img_mious.sort(key=lambda x: x[1], reverse=True)
        top30 = per_img_mious[:50]

        top_csv = os.path.join(base_dir, "top50_miou.csv")
        with open(top_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "image", "mIoU"])
            for rank, (img_name, m) in enumerate(top30, start=1):
                writer.writerow([rank, img_name, round(m, 4)])

        results_dir = base_dir
        os.makedirs(results_dir, exist_ok=True)

        if train_mode == "early_stop":
            csv_path = os.path.join(results_dir, "evaluation_results.csv")
            save_scores_to_csv(csv_path, class_names, class_ious, class_accs, miou, mean_class_acc, pixel_acc)
            print(f"\n=> evaluation_results.csv saved for Round {self.nth_query}\n")

        self._reset_meters()
        return miou

    @staticmethod
    def _query(prob, uncertainty):

        if uncertainty == "least_confidence":
            query = 1.0 - prob.max(dim=1)[0] 

        elif uncertainty == "margin_sampling":
            query = prob.topk(k=2, dim=1).values
            query = (query[:, 0] - query[:, 1]).abs()

        elif uncertainty == "entropy":
            query = (-prob * torch.log(prob + 1e-8)).sum(dim=1)

        elif uncertainty == "random":
            query = torch.rand(prob.shape[0])

        else:
            raise ValueError(f"Unknown query strategy: {uncertainty}")
        H, W = (256, 256)
        N = prob.shape[0] // (H * W)
        query = query.view(N, H, W)
        return query
    
    def _reset_meters(self):
        self.running_loss.reset()
        self.running_score.reset()