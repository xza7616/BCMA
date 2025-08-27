from PIL import Image
import os

# 原始图片所在文件夹路径
input_folder = "/media/wby/shuju/Seg_Water/Under/SUIM/test/image"
# 调整尺寸后输出的新文件夹路径，若不存在则创建它
output_folder = "/media/wby/shuju/Seg_Water/Under/SUIM/try"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历原始文件夹中的所有文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # 打开图片
            image = Image.open(file_path)
            # 调整图片尺寸为480*640，使用Image.ANTIALIAS（高质量下采样滤波）可以让图片缩放后质量更好些
            resized_image = image.resize((480, 640))
            # 构建输出文件的路径
            output_path = os.path.join(output_folder, filename)
            # 保存调整尺寸后的图片到新文件夹
            resized_image.save(output_path)
            print(f"{filename} 图片尺寸调整并保存成功")
        except Exception as e:
            print(f"处理 {filename} 图片时出现错误: {str(e)}")