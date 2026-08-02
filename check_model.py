import MNN

model = "models/lcnet100.mnn"

net = MNN.Interpreter(model)

session = net.createSession()

inputs = net.getSessionInputAll(session)

print("输入:")
print(inputs)

print("模型加载成功")