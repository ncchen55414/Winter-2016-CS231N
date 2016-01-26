import numpy as np

x = np.array([1,2,3,4])
A = np.arange(16).reshape(4,4)

print A
B = np.max(A, axis = 1).reshape(-1,1)
print B




#print y

# z = np.concatenate([y[k] for k in [0,2]])
# print z
# print type(z)

# #Find the most common item in a list
# #Eg.1: Corect output
# lst = [1,1,1,2,2,2]
# ans=max(set(lst), key=lst.count)
# print ans

# #Eg.2: Wrong output 
# lst = [1,1,1,10,10,10]
# ans=max(set(lst), key=lst.count)
# print ans

# lst.sort()
# print lst


########### Test np.sum #################
# A=np.array([[1,2],[3,4]])
# B=np.array([[1,2,3],[4,5,6]])
# #print np.sum(A)
# #print np.sum(A, axis = 0)
# #print np.sum(A, axis = 1)

########### Test Broadcast sum ##########
# A = np.array([1,2,3,4])
# B = np.array([[1],[2],[3]])
# print B.shape
# print A.shape
# print B+A
