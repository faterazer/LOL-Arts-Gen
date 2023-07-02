import os

import torchvision
from PIL import Image

from models.autoencoder import AutoEncoder

autoencoder = AutoEncoder.load_from_checkpoint("./lightning_logs/version_40/checkpoints/epoch=19-val_l2_loss=9501.21.ckpt")
print(autoencoder)
encoder = autoencoder.encoder.eval()
decoder = autoencoder.decoder.eval()

tf = torchvision.transforms.ToTensor()
dir_path = "./LOL-Arts/"
for tag, filename in (("A", "GroupSplashes_VS_RivenYasuo.jpg"), ("B", "VideoStill_odyssey-still-6.jpg")):
    img = Image.open(os.path.join(dir_path, filename))
    x = tf(img).unsqueeze(0).cuda()
    z = encoder(x)
    x_hat = decoder(z)
    torchvision.utils.save_image(x_hat.squeeze(), f"./Temp/temp-example-{tag}.jpg")
