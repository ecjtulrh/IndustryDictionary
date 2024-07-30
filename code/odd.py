def is_odd(num):
    return num % 2 != 0

if __name__ == "__main__":
    while True:
        num = int(input("请输入一个数字： "))
        if is_odd(num):
            print("这是一个奇数")
        else:
            print("这不是一个奇数")
