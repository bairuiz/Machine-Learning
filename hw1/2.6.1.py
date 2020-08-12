def findLucas(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    else:
        return findLucas(n-1) + findLucas(n-2)

print(findLucas(32))
print('Hello World.')