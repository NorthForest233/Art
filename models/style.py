import contextlib
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
import torchvision.transforms.v2 as transforms
import utils
import time
from models.extractors import ContentLoss, StyleLoss
import os


class Art:
    def __init__(self, width, height, content_layers, style_layers, loss_weights, color_fusion, num_augmentations, use_style_mask, use_grad_mask,
                 warmup_epochs, extractor_config, detector_config, device=None):
        self.image_size = (height, width)
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.content_weight = loss_weights['content_weight']
        self.style_weight = loss_weights['style_weight']
        self.variation_weight = loss_weights['variation_weight']
        self.detector_weight = loss_weights['detector_weight']

        self.color_fusion = color_fusion
        self.num_augmentations = num_augmentations
        self.use_style_mask = use_style_mask
        self.use_grad_mask = use_grad_mask
        self.warmup_epochs = warmup_epochs
        self.device = torch.device(
            device or 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.extractor = utils.instantiate_from_config(extractor_config).to(self.device)

        self.detector = utils.instantiate_from_config(detector_config).to(self.device)

        self.transform = nn.Sequential(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(1, 1.2), shear=10)
        )

    def load_image_from_tensor(self, image, size, mode='bilinear'):
        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[-2] != size[0] or image.shape[-1] != size[1]:
            image = F.interpolate(image, size=size, mode=mode)
        return image.contiguous().to(self.device)
    
    def fuse_images(self, content_image, style_image, weight, kernel_size, sigma):
        blurred_content_image = transforms.functional.gaussian_blur(content_image, kernel_size=kernel_size, sigma=sigma)
        blurred_style_image = transforms.functional.gaussian_blur(style_image, kernel_size=kernel_size, sigma=sigma)
        return torch.clamp(style_image - weight * (blurred_style_image - blurred_content_image), 0, 1)

    def set_images(self, content_image, style_image, mask, synthesized_image=None):
        self.content_image = self.load_image_from_tensor(content_image, self.image_size)
        self.style_image = self.load_image_from_tensor(style_image, self.image_size)
        if self.color_fusion['weight']:
            self.style_image = self.fuse_images(self.content_image, self.style_image, **self.color_fusion)
        self.mask = self.load_image_from_tensor(mask, self.image_size)
        self.synthesized_image = nn.Parameter(synthesized_image or self.content_image).to(self.device)

    @torch.no_grad()
    def get_criterions(self, content_tensor, style_tensor):
        content_output = self.extractor(content_tensor)
        style_output = self.extractor(style_tensor)
        content_losses = []
        style_losses = []
        for content_layer in self.content_layers:
            content_losses.append(ContentLoss(content_output[content_layer]))
        for style_layer in self.style_layers:
            if self.use_style_mask and self.use_style_mask:
                style_losses.append(StyleLoss(
                    style_output[style_layer],
                    F.interpolate(self.mask, size=style_output[style_layer].shape[-2:], mode='bilinear')
                ))
            else:
                style_losses.append(StyleLoss(style_output[style_layer]))
        return content_losses, style_losses, lambda x: torch.mean(torch.abs(x))

    @torch.no_grad()
    def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.LBFGS([self.synthesized_image], learning_rate)
        return optimizer

    def train(self, num_epochs, learning_rate, output_dir, output_freq):
        optimizer = self.configure_optimizers(learning_rate)
        content_criterions, style_criterions, variation_criterion = self.get_criterions(
            self.content_image, self.style_image)

        self.postprocess(self.synthesized_image, output_dir, 0)

        for epoch in range(num_epochs):
            start = time.time()

            def closure():
                self.synthesized_image.data.clamp_(0, 1)
                optimizer.zero_grad()

                content_loss = 0
                style_loss = 0
                varation_loss = 0
                detector_loss = 0

                if self.content_weight or self.style_weight:
                    features_output = self.extractor(self.synthesized_image)
                    if self.content_weight:
                        for i, content_criterion in zip(self.content_layers, content_criterions):
                            content_loss += content_criterion(features_output[i])
                    if self.style_weight:
                        for i, style_criterion in zip(self.style_layers, style_criterions):
                            style_loss += style_criterion(features_output[i])
                if self.variation_weight:
                    varation_loss = variation_criterion(self.synthesized_image[:, :, 1:, :] - self.synthesized_image[:, :, :-1, :]) + \
                        variation_criterion(self.synthesized_image[:, :, :, 1:] - self.synthesized_image[:, :, :, :-1])

                if self.detector_weight:
                    if self.num_augmentations == 0:
                        output = self.detector(self.synthesized_image)
                        detector_loss = self.detector.get_loss(output, 1 - self.mask)
                    else:
                        images_list = []
                        masks_list = []
                        for _ in range(self.num_augmentations):
                            augmented_image, augmented_mask = self.transform(
                                (self.synthesized_image, datapoints.Mask(self.mask)))
                            images_list.append(augmented_image)
                            masks_list.append(augmented_mask)
                        augmented_images = torch.concat(images_list)
                        augmented_masks = torch.concat(masks_list)

                        output = self.detector(augmented_images)
                        detector_loss = self.detector.get_loss(output, 1 - augmented_masks)
                        for i in range(self.num_augmentations):
                            with contextlib.suppress():
                                self.save_image(augmented_images[i].detach(), f'./augmentation/image_{i}.jpg')
                                self.save_image(augmented_masks[i].detach(), f'./augmentation/mask_{i}.jpg')

                if epoch < self.warmup_epochs:
                    loss = self.content_weight * content_loss + self.style_weight * style_loss + self.variation_weight * varation_loss
                else:
                    loss = self.content_weight * content_loss + self.style_weight * style_loss + self.variation_weight * varation_loss + \
                        self.detector_weight * detector_loss
                loss.backward()
                if self.use_grad_mask:
                    self.synthesized_image.grad *= self.mask
                return loss

            loss = optimizer.step(closure)
            end = time.time()
            print(f'Epoch: {epoch + 1:2d}, Elapsed time: {end - start:.2f}s, Loss: {loss.data:.2g}')
            self.synthesized_image.data.clamp_(0, 1)

            if (epoch + 1) % output_freq == 0:
                self.postprocess(self.synthesized_image, output_dir, (epoch + 1) // output_freq)

    def save_image(self, x, path):
        img = transforms.functional.to_pil_image(x.squeeze(0))
        img.save(path)

    def postprocess(self, synthesized_image, output_dir, index):
        os.path.exists(output_dir) or os.mkdir(output_dir)
        with torch.no_grad():
            self.save_image(synthesized_image, os.path.join(output_dir, f'image_{index}.jpg'))
            print(f'Save "image_{index}.jpg" successfully!')
            output = self.detector(self.synthesized_image)
            self.save_image(self.detector.get_mask(output), os.path.join(output_dir, f'mask_{index}.jpg'))
            print(f'Save "mask_{index}.jpg" successfully!')
