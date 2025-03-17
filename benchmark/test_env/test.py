my_dict = {0: 10, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 7}
sortedDict = sorted(my_dict.items(),key=lambda x:x[1],reverse=True)
print(sortedDict)