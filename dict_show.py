names = {}

file = open("dataset/name_list.txt", 'r')
lines = file.readlines()
file.close()

for line in lines:
    key, value = line.strip().split("=")
    names[key] = value

print(names["0"])
print(names["1"])