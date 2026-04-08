import segno

n = 1

with open('registry.txt', 'r', encoding='UTF-8') as file:
    for line in file:
        info = line.strip("\n")
        img = segno.make(info, micro=False)
        img.save('qr-codes/' + str(n) + '.png')
        n += 1

