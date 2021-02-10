a = ["a", "b", "c", "d", "e"]

for i, item1 in enumerate(a):
    print(item1)
    for j, item2 in enumerate(a):
        if item2 == "b":
            a.remove(item2)

print(a)
