import qrcode 
import string
import random

characters = string.ascii_letters + string.digits + string.punctuation

for x in range(40):

    random_string = ''.join(random.choices(characters, k=random.randint(4, 128)))
    qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
    qr.add_data(random_string)

    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save('./codes/' +str(x) + 'qr.png')