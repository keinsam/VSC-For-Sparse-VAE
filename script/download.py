from logic.data import load_fashion_mnist, load_mnist


if __name__ == "__main__":
    mnist = load_mnist()
    fashion_mnist = load_fashion_mnist()
    print(len(mnist), len(fashion_mnist))