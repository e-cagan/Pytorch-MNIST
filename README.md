# MNIST PyTorch Example

This repository contains an example implementation of a neural network for classifying handwritten digits from the MNIST dataset using PyTorch.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mnist_pytorch.git
   cd mnist_pytorch
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the training script:

   ```bash
   python train.py
   ```

5. View the results:

   After training, you can visualize the results using:

   ```bash
   python visualize.py
   ```

## Directory Structure

```
mnist_pytorch/
├── LICENSE
├── README.md
├── data_loader.py
├── model.py
├── evaluate.py
├── requirements.txt
├── train.py
└── main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project is inspired by the PyTorch official tutorials and the MNIST dataset provided by Yann LeCun.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## Contact
For any questions or feedback, please open an issue on the GitHub repository or contact the author at
[
    emincaganapaydin@gmail.com
].