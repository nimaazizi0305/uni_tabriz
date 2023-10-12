# list_of_item=[5,7,4,-3,9,1,10,4,5,8,9,3]
# i=0
# number_of_item=[1]
# length_of_list=len(list_of_item)

# for num in list_of_item:
#     i=i+1
#     if i>= length_of_list:
#         break
#     elif list_of_item[i] > list_of_item[i-1]:
#         len_number_of_item=len(number_of_item)
#         next_index=len_number_of_item+1
#         max_number=max(number_of_item)
#         next_index=max_number+1
#         number_of_item.append(next_index)
#     elif list_of_item[i] < list_of_item[i-1]:
#         next_index=1
#         number_of_item.append(next_index)

# print(number_of_item)




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



















