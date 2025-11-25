import pikepdf

# 打开两个PDF文件
pdf1 = pikepdf.open('/home/xkj/project/KRAS/plots/附图5_1.pdf')
pdf2 = pikepdf.open('/home/xkj/project/PIK3CA/plots/附图5_2.pdf')

# 创建一个新的PDF文件
output = pikepdf.Pdf.new()

# 将两个PDF文件的页面合并
output.pages.extend(pdf1.pages)
output.pages.extend(pdf2.pages)

# 保存合并后的PDF
output.save('附图5.pdf')

print("PDF合并成功！输出文件为: 附图5.pdf")
