def print_info():
    print("-" * 20)
    print("欢迎登录学生管理系统!")
    print("1.添加学员")
    print("2.删除学员")
    print("3.修改学员信息")
    print("4.查询学员信息")
    print("5.显示所有学员信息")
    print("6.退出系统")
    print("-" * 20)


info = []


def add_info():
    """添加学员信息"""
    id = input("请输入学号:")
    name = input("请输入姓名:")
    phone = input("请输入手机号:")

    for stu in info:
        if stu["name"] == name:
            print("学生已经存在请重新选择!")
            return

    stuDict = {}
    stuDict["name"] = name
    stuDict["id"] = id
    stuDict["phone"] = phone

    info.append(stuDict)
    print(info)


def del_info():
    """删除学员信息"""
    name = input("请输入要删除的学员姓名:")
    for stu in info:
        if stu["name"] == name:
            info.remove(stu)
            print("删除学员 %s 成功!" % name)
            return
    print("未找到学员 %s " % name)


try:
    print("xxxxx")
except EOFError:
    print_info()

while True:
    print_info()
    selectNum = input("请选择要进行的操作:")

    if selectNum == "1":
        print("添加学员")
        add_info()
    elif selectNum == "2":
        print("删除学员")
        del_info()
    elif selectNum == "3":
        print("修改学员信息")
    elif selectNum == "4":
        print("查询学员信息")
    elif selectNum == "5":
        print("显示所有学员信息")
    elif selectNum == "6":
        print("退出系统")
        break
    else:
        print("不支持的操作!")
