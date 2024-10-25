from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import config
from generator import Generator


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(config.DEVICE)

def load_generator_checkpoint(checkpoint_path, gen_model, optimizer=None, lr=None):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    gen_model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return gen_model


def generate_zebra_image(horse_image_path, gen_Z, transform):
    horse_image = preprocess_image(horse_image_path, transform)  # Preprocess horse image

    with torch.no_grad():  # We don't need gradients for inference
        fake_zebra = gen_Z(horse_image)  # Generate the zebra image

    return fake_zebra


def save_output_image(tensor, save_path):
    # Post-process the output: Convert from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5  # Rescale to [0, 1]
    tensor = tensor.squeeze(0)  # Remove batch dimension
    save_image(tensor, save_path)  # Save the image


def horse_to_zebra_inference(horse_image_path, save_path):
    # Load the generator
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Z = load_generator_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z)
    gen_Z.eval()  # Set the generator to eval mode

    # Define the preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess horse image, generate zebra, and save output
    fake_zebra = generate_zebra_image(horse_image_path, gen_Z, transform)
    save_output_image(fake_zebra, save_path)



if __name__ == "__main__":
    horse_image_path = "/Users/manas/Documents/GitHub/Cycle-GAN/CamouflageGeneration/dataset/horsezebra/testA/n02381460_530.jpg"  # Path to the horse image
    save_path = "horse_output.jpg"  # Path to save the zebra image
    horse_to_zebra_inference(horse_image_path, save_path)