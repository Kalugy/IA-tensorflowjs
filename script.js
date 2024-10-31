/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
 
//import {TRAINING_DATA2} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';



// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;


// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;


// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);


// Input feature Array is 1 dimensional.

const INPUTS_TENSOR = tf.tensor2d(INPUTS);


// Output feature Array is 1 dimensional.

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);


// Now actually create and define model architecture.

const model = tf.sequential();


model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

model.add(tf.layers.dense({units: 16, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();


train();


async function train() { 

  // Compile the model with the defined optimizer and specify our loss function to use.

  model.compile({

    optimizer: 'adam',

    loss: 'categoricalCrossentropy',

    metrics: ['accuracy']

  });


  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {

    shuffle: true,        // Ensure data is shuffled again before using each epoch.

    validationSplit: 0.2,

    batchSize: 512,       // Update weights after every 512 examples.      

    epochs: 50,           // Go over the data 50 times!


  });

  

  OUTPUTS_TENSOR.dispose();

  INPUTS_TENSOR.dispose();

  evaluate(); // Once trained we can evaluate the model.

}

const PREDICTION_ELEMENT = document.getElementById('prediction');


function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs. 

 

  let answer = tf.tidy(function() {

    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    

    let output = model.predict(newInput);

    output.print();

    return output.squeeze().argMax();    

  });
   answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = index;

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}

const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28);

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }


  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 


  // Perform a new classification after a certain interval.

  setTimeout(evaluate, 2000);

}




