#一次跑完全部
import subprocess
import threading
import sys
import logging
import time

start_time=time.time()

# 設置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('all_output.txt'),
        logging.StreamHandler()  # 同時輸出到控制台
    ]
)

current_python = sys.executable
logging.info(f"使用 Python 解釋器: {current_python}")

def read_output(pipe, prefix=''):
    for line in iter(pipe.readline, ''):
        # 移除行尾可能有的換行符，並加上前綴
        line = line.rstrip('\n')
        logging.info(f"{prefix}{line}")
    pipe.close()


# 定義要傳遞的參數
epoch = [20,40,80]
bs = [8,16,32]
lr = [0.1,0.01,0.001]


r=[epoch,bs,lr]

logging.info("HW4-1")
logging.info("       ")
for i in range(3):
    for j in range(3):
        
        lstart_time=time.time()
        
        e=r[i][j] if i==0 else r[0][0]
        b=r[i][j] if i==1 else r[1][2]
        l=r[i][j] if i==2 else r[2][1]
        logging.info(f"以下是由epoch {e} ,batch_size {b} ,learning_rate {l} 的作法:")
        
        #train
        process = subprocess.Popen(
            [current_python, "train.py", '1', str(e), str(b), str(l)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))



        stdout_thread.start()
        stderr_thread.start()


        stdout_thread.join()
        stderr_thread.join()


        return_code = process.wait()
        logging.info(f"train執行完成， {return_code}")
        
        
        #test
        process = subprocess.Popen(
            [current_python, "test.py", '1', str(e), str(b), str(l)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))



        stdout_thread.start()
        stderr_thread.start()


        stdout_thread.join()
        stderr_thread.join()


        return_code = process.wait()
        logging.info(f"test執行完成， {return_code}")
        lend_time=time.time()
        time_diff = lend_time - lstart_time


        hours = int(time_diff // 3600)
        minutes = int((time_diff % 3600) // 60)
        seconds = int(time_diff % 60)
        formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        logging.info(f"耗時: {formatted_time}")  # 例如: 00:01:23

        
        
        
logging.info("HW4-1執行完成")
logging.info("===================================")
logging.info("以下是HW4-2")
e=r[0][0]
b=r[1][2]
l=r[2][1]
logging.info(f"是由epoch{e},batch_size{b},learning_rate{l}的作法:")

for i in range(5):
    #train    方法=[nn.MSELoss(),nn.KLDivLoss(),nn.HingeEmbeddingLoss()]
    process = subprocess.Popen(
        [current_python, "train.py", '2', str(e), str(b), str(l), str(i)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
        
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))



    stdout_thread.start()
    stderr_thread.start()


    stdout_thread.join()
    stderr_thread.join()


    return_code = process.wait()
    logging.info(f"train執行完成， {return_code}")
    
    
    #test
    process = subprocess.Popen(
        [current_python, "test.py", '2', str(e), str(b), str(l)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
        
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))



    stdout_thread.start()
    stderr_thread.start()


    stdout_thread.join()
    stderr_thread.join()


    return_code = process.wait()
    logging.info(f"test執行完成， {return_code}")
    

end_time=time.time()
time_diff = end_time - start_time


hours = int(time_diff // 3600)
minutes = int((time_diff % 3600) // 60)
seconds = int(time_diff % 60)


formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
logging.info(f"總耗時: {formatted_time}")  # 例如: 00:01:23
logging.info("HW4執行完成")