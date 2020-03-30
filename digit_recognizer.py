from digit_network import NeuralNetwork
from digit_matrix import Matrix
import math, random, time, turtle
import csv

wn = turtle.Screen()
wn.setup(560, 560)
wn.title('Digit Reconizer Interface')
wn.tracer(0)
wn.colormode(255)

width = wn.window_width()
height = wn.window_height()


xs = []
ys = []
for i in range(int(-width/2 - 1), int(height/2)):
    if i % 10 == 0:
        xs.append(i)
        ys.append(i)
pixelPositions = []
xcount = 0
ycount = 0
for y in range(len(ys), 0, -1):
    for x in range(len(xs)):
        if xcount % 2 == 1 and ycount % 2 == 1:
            pixelPositions.append((xs[x], ys[y]))
        xcount += 1
    ycount += 1

pixels = []

for i in range(len(pixelPositions)):
    p = turtle.Turtle()
    p.speed(0)
    p.shape('square')
    p.color('black', 'white')
    p.pu()
    p.goto(pixelPositions[i])
    pixels.append(p)

def show(colors, numPredicted):
    for i in range(len(pixels)):
        p = pixels[i]
        c = int(255 - (colors[i] * 255))
        p.color(c, c, c)
    w.clear()
    w.write('AI Prediction: ' + str(numPredicted), move=False, align="left", font=("Courier", 10, "bold"))
    wn.update()

mouse = turtle.Turtle()
mouse.speed(0)
mouse.pu()
mouse.color('black', 'white')
mouse.left(120)

def draw(x, y):
    mouse.goto(x, y)
    for p in pixels:
        if p.distance(mouse) < 20:
            p.color('black')
        elif p.distance(mouse) < 25 and p.pencolor() == 'black':
            c = 255 - int(p.distance(mouse) * 10)
            # print(c)
            p.color(c, c, c)

def clearAll():
    for p in pixels:
        p.color('black', 'white')

wn.listen()


while True:
    mouse.ondrag(draw)
    wn.onscreenclick(mouse.goto)
    wn.onkey(clearAll, 'space')
    wn.update()
    break

def greatestOf(list):
    greatest = -1
    for val in list:
        if val > greatest:
            greatest = val
    # return greatest
    return list.index(greatest)

w = turtle.Turtle()
w.speed(0)
w.pu(); w.ht()
w.goto(-width/2 + 10, -height/2 + 10)


main = []
nn = NeuralNetwork(784, 16, 10)

correct = 0
total = 0
bestAccuracy = -1

errors = {}
done = []
for i in range(10):
    for j in range(10):
        if i != j and f'{j} {i}' not in done:
            errors[f'{i} {j}'] = 0
            done.append(f'{i} {j}')

def Error():
    global errors
    print(errors)


wn.onkey(Error, 'space')
wn.onclick(mouse.goto)

input('Start Learning: \n')

wn.bye()

s = open('submission.csv', 'w+')
s.write('ImageId,Label\n')
s.close()

# Reading Data File
with open('train.csv') as file:

    # Getting CSV file
    reader = csv.DictReader(file)
    rows = 0

    # Reading CSV file
    for row in reader:

        # Initializing Row Array
        main.append([])

        # Extracting Data
        last = main[len(main) - 1]
        for i in range(784):
            pixel = row['pixel' + str(i)]
            # Normalizing Data -> [0 - 1]
            last.append((float(pixel) / 255.0))

        # Getting Answer
        target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        target[int(row['label'])] = 1

        # Training Model
        nn.train(last, target)

        # Getting prediction
        numPredicted = int(greatestOf(nn.predict(last)))
        numAnswer = target.index(1)

        # Showing AI Guess
        # show(last, numPredicted)

        # Recording Results
        if numPredicted == numAnswer:
            correct += 1
        else:
            # Recording Errors
            string = f'{numPredicted} {numAnswer}'
            if string in done:
                errors[string] += 1
            else: errors[f'{numAnswer} {numPredicted}'] += 1
        # Calculating Accuracy
        total += 1
        accuracy = ((correct/total) * 100)
        if int(accuracy) > int(bestAccuracy):
            bestAccuracy = accuracy
            # Printing Data
            if total % 1 == 0:
                print(f"Accuracy: {int(accuracy)}%")
        
        if total % 1000 == 0:
            print(f'\n<<<<< {total} finished >>>>>\n')

        # Ending Learning (Lag Reduction)
        # rows += 1
        # if rows > 20000: break


print(f"Correct: {(correct/total) * 100}%")


input('done')
wn.mainloop()