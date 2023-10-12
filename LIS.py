def lis(arr):
    n = len(arr)

    lis = [1]*n
 
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j]+1
 

    maximum = 0
 
    for i in range(n):
        maximum = max(maximum, lis[i])
 
    return maximum
 
print(lis([5,7,4,-3,9,1,10,4,5,8,9,3]))



















