# Importing important imports
from digit_network import NeuralNetwork
import random, time
import turtle, csv

# Interface Window Setup
wn = turtle.Screen()
wn.setup(560, 560)
wn.title('Digit Reconizer Interface')
wn.tracer(0)
wn.colormode(255)

# Dimensions of window
width = wn.window_width()
height = wn.window_height()

# Setting up grid
xs = []
ys = []
for i in range(int(-width/2 - 1), int(height/2)):
    if i % 10 == 0:
        xs.append(i)
        ys.append(i)
pixelPositions = []
xcount = 0
ycount = 0
# Main array
for y in range(len(ys), 0, -1):
    for x in range(len(xs)):
        if xcount % 2 == 1 and ycount % 2 == 1:
            pixelPositions.append((xs[x], ys[y]))
        xcount += 1
    ycount += 1

# Setting up window pixels
pixels = []
for i in range(len(pixelPositions)):
    p = turtle.Turtle()
    p.speed(0)
    p.shape('square')
    p.color('black', 'white')
    p.pu()
    p.goto(pixelPositions[i])
    pixels.append(p)

# Showing inputs in interface
def show(colors, numPredicted):
    for i in range(len(pixels)):
        p = pixels[i] * 255
        c = int(255 - (colors[i] * 255))
        p.color(c, c, c)
    w.clear()
    w.write('AI Prediction: ' + str(numPredicted), move=False, align="left", font=("Courier", 10, "bold"))
    wn.update()

# Manual Drawing input
mouse = turtle.Turtle()
mouse.speed(0)
mouse.pu()
mouse.color('black', 'white')
mouse.left(120)

# Drawing functions
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

# Draw Loop
while True:
    mouse.ondrag(draw)
    wn.onscreenclick(mouse.goto)
    wn.onkey(clearAll, 'space')
    wn.update()
    break

# Getting greatest value of List
def greatestOf(list):
    greatest = -1
    for val in list:
        if val > greatest:
            greatest = val
    # return greatest
    return list.index(greatest)

# AI Prediction Writer
w = turtle.Turtle()
w.speed(0)
w.pu(); w.ht()
w.goto(-width/2 + 10, -height/2 + 10)

# Main Neural Network and Inputs Setup
pixelInputs = []
nn = NeuralNetwork(784, 16, 10)

# Data Initialization
correct = 0
total = 0
bestAccuracy = -1

# Error Initialization
errors = {}
done = []
for i in range(10):
    for j in range(10):
        if i != j and f'{j} {i}' not in done:
            errors[f'{i} {j}'] = 0
            done.append(f'{i} {j}')

# Printing Error Function
def Error():
    global errors
    print(errors)

# Hotkeying Error Functions
wn.onkey(Error, 'space')
wn.onclick(mouse.goto)

# Starting Learning
input('Start Learning: \n')

# Exiting Window (Lag Reduction)
wn.bye()

# Initializing Main Submission File
s = open('submission.csv', 'w+')
s.write('ImageId,Label\n')
s.close()

# Starting Timer
start_time = time.time()

# Reading Data File
with open('train.csv') as file:

    # Getting CSV file
    reader = csv.DictReader(file)

    # Reading CSV file
    for row in reader:

        # Initializing Row Array
        pixelInputs.append([])

        # Extracting Data
        last = pixelInputs[len(pixelInputs) - 1]
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
            print(f"Accuracy: {int(accuracy)}%")
        
        if total % 1000 == 0:
            print(f'\n<<<<< {total} finished >>>>>\n')

        # Ending Learning (Lag Reduction)
        # if total > 2000: break

# ==========================================================================

# Printing data
end_time = time.time()
input(f'Learning Completed\nAccuracy: {(correct/total) * 100}%\nElapsed Time: {end_time - start_time} seconds\n\nProceed to final tests: \n')

# Opening Submission File
submit = open('submission.csv', 'w+')
submit.write('ImageId, Label\n')

# Writing in Submission
with open('test.csv') as file:

    # Opening File
    reader = csv.DictReader(file)
    index = 1

    # Reading through Rows of Testing
    for row in reader:

        # Reading Pixel Columns into Array
        inputs = []
        for i in range(784):
            pixel = row['pixel' + str(i)]
            inputs.append((float(pixel) / 255.0))
        
        # Feedforward Algorithm Prediction
        guess = nn.predict(inputs)
        predicted = greatestOf(guess)

        # Dumping data to submission file
        submit.write(str(index) + ',' + str(predicted) + '\n')

        # Updating Console
        index += 1
        if index % 1000 == 0:
            print('Index:', index)

# Closing and updating file
submit.close()


# Printing Data
print('\n\nDATA DUMP COMPLETE: \'submission.csv\' is ready for submit\n')

# Reading Neural Network to file
brain = open('brain.txt', 'w+')
brain.write(f"{nn.weights_ih.data}\n{nn.weights_ho.data}\n{nn.bias_h.data}\n{nn.bias_o.data}")
brain.close()

# Printing Data
print(
    f"{nn.weights_ih.data}\n{nn.weights_ho.data}\n{nn.bias_h.data}\n{nn.bias_o.data}"
)

# Ending program
input('\n\nEnd Program: ')
# wn.mainloop()
