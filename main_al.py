from args import Arguments
from models.model import Model

def main(args):
    Model(args)()

if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)