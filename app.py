from flask import Flask, render_template, request, send_file
from PIL import Image as PILImage
from upscaler import txsal, lanczos, cubic, area, FSRCNN, ESPCN, SRN
import io

app = Flask(__name__)

class ImageUpscalerApp:
        
    def __init__(self):
        self.image_path = ""
        self.upscaling_method = "2xSal Scaling"
        self.upscaled_image = None

    def upscale_image(self):
        if not self.image_path:
            return None
        try:
            source_image = PILImage.open(self.image_path)
            if self.upscaling_method == '2xSal Scaling':
                upscaled_image = txsal(source_image)
            elif self.upscaling_method == 'Lanczos interpolation':
                upscaled_image = lanczos(source_image)
            elif self.upscaling_method == 'Cubic interpolation':
                upscaled_image = cubic(source_image)
            elif self.upscaling_method == 'Area interpolation':
                upscaled_image = area(source_image)
            elif self.upscaling_method == 'FSRCNN':
                upscaled_image = FSRCNN(source_image)
            elif self.upscaling_method == 'ESPCN':
                upscaled_image = ESPCN(source_image)
            elif self.upscaling_method == 'LapSRN':
                upscaled_image = SRN(source_image)
            else:
                raise ValueError(f"Invalid method '{self.upscaling_method}' selected.")
            output_buffer = io.BytesIO()
            upscaled_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            return output_buffer
        except Exception as e:
            print(f"Error: {e}")
            return None

image_upscaler = ImageUpscalerApp()
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_upscaler.image_path = image
            image_upscaler.upscaling_method = request.form['method']
            upscaled_image_buffer = image_upscaler.upscale_image()
            if upscaled_image_buffer:
                return send_file(upscaled_image_buffer, mimetype='image/png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
