def draw(depth, height, width):

    for i in range(depth-1):
        print((depth - i - 1) * ' ' + width * '_' + ' ' + i * '|')

    for j in range(height):
        
        shape = + width * 'X' + ' ' 
        dep = min(depth-1, (height - j))* '|'
        print(shape + dep)
