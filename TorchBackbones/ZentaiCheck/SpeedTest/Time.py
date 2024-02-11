import time

print('我是time()方法：{}'.format(time.time()))
print('我是perf_counter()方法：{}'.format(time.perf_counter()))
print('我是process_time()方法：{}'.format(time.process_time()))
t0 = time.time()
c0 = time.perf_counter()
p0 = time.process_time()
r = 0
for i in range(10000000):
    r += i
time.sleep(2)
print(r)
t1 = time.time()
c1 = time.perf_counter()
p1 = time.process_time()
spend1 = t1 - t0
spend2 = c1 - c0
spend3 = p1 - p0
print("time()方法用时：{}s".format(spend1))
print("perf_counter()用时：{}s".format(spend2))
print("process_time()用时：{}s".format(spend3))
print("测试完毕")
