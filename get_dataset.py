# 本代码用于从xeno-canto上获取指定物种的音频
import os
import requests


# 创建目录
def create_directory(directory):
    os.makedirs(directory)


create_directory('/kaggle/working/Coccothraustes_coccothraustes')

# 设置API端点和参数
api_endpoint = 'https://xeno-canto.org/api/2/recordings?query=Coccothraustes%20coccothraustes'

# 发送GET请求获取数据
response = requests.get(api_endpoint)

# 处理响应数据
data = response.json()
print("查找成功！")
print(f"共{data['numRecordings']}条数据")

# 下载音频文件
cnt = 0
for record in data['recordings']:
    try:
        cnt += 1
        file_url = record['file']  # 获取音频文件链接
        file_name = record['file-name'][:8] + record['file-name'][-4:]  # 使用链接中的文件名作为本地文件名
        # 下载文件
        response = requests.get(file_url)
        with open(f'/kaggle/working/Coccothraustes_coccothraustes/{file_name}', 'wb') as file:
            file.write(response.content)
        print(f'{file_name} 下载完成')
    except:
        cnt -= 1
        print(f"{record['file']} 无法下载")

    if cnt >= 510:
        break

print('所有音频文件下载完成')
