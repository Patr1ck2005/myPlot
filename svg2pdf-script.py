import os
import subprocess


def convert_svg_to_pdf(svg_file):
    # 获取原文件的目录和文件名
    file_dir = os.path.dirname(svg_file)
    file_name = os.path.splitext(os.path.basename(svg_file))[0]

    # 生成 PDF 文件路径，保存在原目录下
    pdf_file = os.path.join(file_dir, file_name + '.pdf')

    # 使用 Inkscape 命令行工具转换 SVG 为 PDF
    command = ['E:\\Program Files\\Inkscape\\bin\\inkscape.exe', svg_file, '--export-filename', pdf_file]
    subprocess.run(command)


def convert_svgs_in_directory(input_dir):
    # 遍历目录下的所有 SVG 文件并转换
    for filename in os.listdir(input_dir):
        if filename.endswith('.svg'):
            svg_path = os.path.join(input_dir, filename)
            convert_svg_to_pdf(svg_path)
            print(f"已转换：{filename}")


def main(working_directory):

    # 执行转换
    convert_svgs_in_directory(working_directory)

    print("转换完成！")

if __name__ == '__main__':
    # main(r'D:\DELL\Documents\svgs')
    main(r'D:\DELL\Documents\myPlots\examples')
    # main(r'D:\DELL\Documents\myPlots\examples\SM')

