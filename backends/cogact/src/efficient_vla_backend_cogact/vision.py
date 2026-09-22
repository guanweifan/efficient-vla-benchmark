"""Native policy-space RGB for geometry, without tensor normalization round trips."""

from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor


def policy_image(vision, image):
    transforms = vision.image_transform.siglip_image_transform.transforms
    spatial = []
    for transform in transforms:
        if isinstance(transform, ToTensor):
            break
        if not isinstance(transform, (Resize, CenterCrop)):
            raise ValueError("Unsupported native spatial image transform for patch inspection")
        spatial.append(transform)
    else:
        raise ValueError("Native image pipeline has no supported tensor boundary")
    remaining = transforms[len(spatial) :]
    if len(remaining) != 2 or not isinstance(remaining[1], Normalize):
        raise ValueError("Unexpected native transforms after the policy RGB boundary")
    for transform in spatial:
        image = transform(image)
    if image.size != (224, 224):
        raise ValueError("Native patch inspection requires a 224x224 policy image")
    return image
