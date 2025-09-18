a,b=input(),input()
if ',' in a:
    a=a.replace(',','.')
if ',' in b:
    b=b.replace(',','.')
print(a,b)
print(f'a: {float(a)}\nb: {float(b)}\nsum: {round(float(a)+float(b),2)}\navg: {round((float(a)+float(b))/2,2)}')