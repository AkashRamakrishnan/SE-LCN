# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datasets import train_segmentation_data

def test_custom_dataset(dataset, idx):
    # Get sample from dataset
    image, mask = dataset[idx]

    # Define a function to display the image and mask
    def display_image_mask(image, mask):
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')

        plt.show()

    # Display the image and mask
    display_image_mask(image, mask)

# Example usage:
# dataset = CustomDataset(txt_file='your_txt_file.txt', transform=transforms.Compose([transforms.ToTensor()]))
# test_custom_dataset(dataset)
dataset = train_segmentation_data('/home/s3075451/', 'HAM10000_seg_train.txt')
test_custom_dataset(dataset, 2)