from PIL import Image
import os

# 定义输入和输出目录路径
input_dir = r'.\Dataset\TrainingSet\Pseudo-label\Imgs\\'
output_dir = r'.\Dataset_CLAH\TrainingSet\Pseudo-label\Imgs\\'

# 遍历输入目录下的所有图片文件
for file_name in os.listdir(input_dir):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # 构建输入图像的完整路径
        input_path = os.path.join(input_dir, file_name)

        # 打开图像
        image = Image.open(input_path)

        # 进行自适应直方图增强
        enhanced_image = image.equalize()

        # 构建输出图像的完整路径
        output_path = os.path.join(output_dir, file_name)

        # 保存增强后的图像
        enhanced_image.save(output_path)
