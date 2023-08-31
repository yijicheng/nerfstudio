import os
import os.path as osp

# 创建一个包含图片的HTML表格  
head_html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
  <style>
    table {{
        border-collapse: collapse;  
        width: 50%;  
    }}
    th, td {{
        border: 1px solid black;  
        padding: 8px;  
        text-align: left;  
    }}
  </style>
</head>
<body>

<table>
'''

tail_html = f'''
</table>

</body>
</html>
'''


with open("/mnt/blob/data/rodin/data_10persons.txt", 'r') as f:
    SEEN_SUBJECT_10 = f.read().splitlines()

with open("/mnt/blob/data/rodin/person_test_10.txt", 'r') as f:
    UNSEEN_SUBJECT = f.read().splitlines()

if __name__ == '__main__':

    subject_names = SEEN_SUBJECT_10 + UNSEEN_SUBJECT
    image_dirs = [
        
        "images/eval_stage2_parallel_subject_fitting_st1-30ep_5000it_10ddp_fp32_scale_up_subject",
        "images/eval_single_subject_120k-iter",
    ]
    # row0
    row0 = ["subject_name"]
    for image_dir in image_dirs:
        for image_type in ["img", "depth", "accumulation"]:
          row0.append(image_dir + ":" + image_type)

    # row_1...n
    rows = []
    for subject_name in subject_names:
        row = [subject_name]
        for head_name in row0[1:]:
            image_dir = head_name.split(":")[0]
            image_type = head_name.split(":")[1]
            image_path = osp.join(image_dir, subject_name, f"000000-{image_type}.jpg")
            if image_type == "img":
                width, height = 1024, 512
            else:
                width, height = 512, 512
            row.append(f'<img src="{image_path}" alt="Image" width="{width}" height="{height}">')
        rows.append(row)

    row0_html = """
      <tr>
    """
    for row0_item in row0:
        row0_html += f"""
            <th>{row0_item}</th>
        """
    row0_html += f"""
        </tr>
    """

    rows_html = """

    """
    for row in rows:
        rows_html += f"""
          <tr>
        """
        for row_item in row:
          rows_html += f"""
              <td>{row_item}</td>
          """
        rows_html += f"""
          </tr>
        """

    html = head_html + row0_html + rows_html + tail_html
    # 将HTML代码保存到文件中  
    with open('htmls/rodinhd.html', 'w') as f:  
        f.write(html)  