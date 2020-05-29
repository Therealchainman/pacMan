import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dp = self.run(x)
        scalar = nn.as_scalar(dp)
        return 1 if scalar >= 0 else -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        for _ in range(20):
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)
                if pred == nn.as_scalar(y):
                    continue
                self.w.update(x, nn.as_scalar(y))

class RegressionModel(object):
    """                       
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.number_hidden_nodes = 60
        self.w = [nn.Parameter(1, self.number_hidden_nodes), nn.Parameter(self.number_hidden_nodes, 1)]
        self.b = [nn.Parameter(1, self.number_hidden_nodes), nn.Parameter(1 , 1)]
        self.nn = len(self.w)
        self.lr = -0.03
        self.bs = 50

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Linear combination of first weights and inputs before going to first hidden layer. 
        xn = nn.Linear(x, self.w[0])
        xb = nn.AddBias(xn, self.b[0])
        # the activation function being applied in the hidden layer that introduces nonlinearity. 
        xn = nn.ReLU(xb)
        # Need to make the values back linear for the output with the weights of the output
        xw = nn.Linear(xn, self.w[1])
        pred_y = nn.AddBias(xw, self.b[1])
        return pred_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred_y = self.run(x)
        loss = nn.SquareLoss(pred_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Runs infinite iterations through the dataset with given batch size for each run. 
        for x, y in dataset.iterate_forever(self.bs):
            # Compute the loss for the current batch of data from the dataset. 
            loss = self.get_loss(x, y)
            grad = nn.gradients(loss, [*self.w, *self.b])
            self.w[1].update(grad[1], self.lr)
            self.b[1].update(grad[3], self.lr)
            self.w[0].update(grad[0], self.lr)
            self.b[0].update(grad[2], self.lr)
            # Checks that the loss after updating the weights from above is within the constraint. 
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) <= 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        number_hidden_nodes = 200
        self.bs = 100
        self.num_features = 10
        self.dim = 784
        self.w = [nn.Parameter(self.dim, number_hidden_nodes), nn.Parameter(number_hidden_nodes, self.num_features)]
        self.b = [nn.Parameter(1, number_hidden_nodes), nn.Parameter(1 , self.num_features)]
        self.nn = len(self.w)
        self.lr = -0.5

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x1 = nn.Linear(x, self.w[0])
        x1 = nn.AddBias(x1, self.b[0])
        activate_x = nn.ReLU(x1)
        x2 = nn.Linear(activate_x, self.w[1])
        pred_y = nn.AddBias(x2, self.b[1])
        return pred_y


        """
        To write the hidden layer I'm going to use the activation function. on the inputs.  But I believe 
        you first should have the weights involved. 
        """

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True: 
            for x, y in dataset.iterate_once(self.bs):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [*self.w, *self.b])
                self.w[1].update(grad[1], self.lr)
                self.b[1].update(grad[3], self.lr)
                self.w[0].update(grad[0], self.lr)
                self.b[0].update(grad[2], self.lr)
            acc = dataset.get_validation_accuracy()
            if acc >= 0.975:
                return



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.num_outputs = len(self.languages)
        self.hd = 200
        self.lr = -0.015
        self.bs = 5
        self.w_input = nn.Parameter(self.num_chars, self.hd)
        self.b_input = nn.Parameter(1, self.hd)
        self.w_hidden = nn.Parameter(self.hd, self.hd)
        # self.b_hidden = nn.Parameter(1, self.hd)
        self.w_output = nn.Parameter(self.hd, self.num_outputs)
        self.b_output = nn.Parameter(1, self.num_outputs)
        # print(self.w_input.data, self.b_input.data, self.w_hidden, self.b_hidden, self.w_output, self.b_output)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        initial = True
        # print(xs)
        for char in xs:
            if initial:
                lin = nn.Linear(char, self.w_input)
                bias = nn.AddBias(lin, self.b_input)
                act = nn.ReLU(bias)
                h = act
                initial = False
            else:
                z = nn.Add(nn.Linear(char, self.w_input), nn.Linear(h, self.w_hidden))
                # bias_hidden = nn.AddBias(z, self.b_hidden)
                h = z
        output = nn.Linear(h, self.w_output)
        output = nn.AddBias(output, self.b_output)
        return output

        

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred_y = self.run(xs)
        loss = nn.SoftmaxLoss(pred_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True: 
            for x, y in dataset.iterate_once(self.bs):
                # print(f"({x[0].data},{y})")
                loss = self.get_loss(x, y)
                # print(loss.data)
                grad = nn.gradients(loss,[self.w_input, self.b_input, self.w_hidden, self.w_output, self.b_output])
                # print(grad[0].data,grad[1].data,grad[2].data,grad[3].data,grad[4].data,grad[5].data)  
                self.w_input.update(grad[0], self.lr)
                self.b_input.update(grad[1], self.lr)
                self.w_hidden.update(grad[2], self.lr)
                # self.b_hidden.update(grad[3], self.lr)
                self.w_output.update(grad[3], self.lr)
                self.b_output.update(grad[4], self.lr)
            acc = dataset.get_validation_accuracy()
            if acc >= 0.82:
                return