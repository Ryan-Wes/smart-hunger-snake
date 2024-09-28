const playBoard = document.querySelector(".play-board");
const scoreElement = document.querySelector(".score");
const highScoreElement = document.querySelector(".high-score");

let gameOver = false;
let foodX, foodY;
let snakeX = 5, snakeY = 10;
let snakeBody = [];
let velocityX = 0, velocityY = 0;
let setIntervalId;
let score = 0;

let highScore = localStorage.getItem("high-score") || 0;
highScoreElement.innerText = `High Score: ${highScore}`;

const experiences = [];

const getState = () => {
    return {
        head: [snakeX, snakeY],
        food: [foodX, foodY],
        body: snakeBody,
        direction: [velocityX, velocityY]
    };
};

const actions = {
    up: 0,
    down: 1,
    left: 2,
    right: 3
};

const getReward = () => {
    if (collisionWithBody() || snakeX <= 0 || snakeX > 30 || snakeY <= 0 || snakeY > 30) {
        return -50; // Penalidade maior
    }
    if (snakeX === foodX && snakeY === foodY) {
        return 10; // Comer comida
    }
    return -1; // Penalidade leve
};

const collisionWithBody = () => {
    for (let i = 1; i < snakeBody.length; i++) {
        if (snakeBody[i][0] === snakeX && snakeBody[i][1] === snakeY) {
            return true;
        }
    }
    return false;
};

const model = tf.sequential();
model.add(tf.layers.dense({ units: 48, activation: 'relu', inputShape: [6] }));
model.add(tf.layers.dense({ units: 48, activation: 'relu' }));
model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

const trainModel = async (batchSize = 32) => {
    if (experiences.length < batchSize) return;

    const batch = experiences.slice(0, batchSize);
    const inputs = batch.map(exp => [exp.state.head[0], exp.state.head[1], exp.state.food[0], exp.state.food[1], exp.state.direction[0], exp.state.direction[1]]);
    const rewards = batch.map(exp => exp.reward);

    const inputTensor = tf.tensor2d(inputs, [batchSize, 6]);
    const rewardTensor = tf.tensor2d(rewards, [batchSize, 1]);

    await model.fit(inputTensor, rewardTensor);
};

const changeFoodPosition = () => {
    foodX = Math.floor(Math.random() * 30) + 1;
    foodY = Math.floor(Math.random() * 30) + 1;
};

const handleGameOver = () => {
    clearInterval(setIntervalId);
    alert("Game Over! Press Ok to replay...");
    location.reload();
};

const chooseAction = () => {
    const state = getState();
    const stateTensor = tf.tensor2d([[state.head[0], state.head[1], state.food[0], state.food[1], velocityX, velocityY]]);
    const actionValues = model.predict(stateTensor).dataSync();
    const action = actionValues.indexOf(Math.max(...actionValues)); 

    // Adicionando aleatoriedade
    if (Math.random() < 0.1) {
        return Math.floor(Math.random() * 4);
    }

    switch (action) {
        case actions.up:
            if (velocityY != 1) {
                velocityX = 0;
                velocityY = -1;
            }
            break;
        case actions.down:
            if (velocityY != -1) {
                velocityX = 0;
                velocityY = 1;
            }
            break;
        case actions.left:
            if (velocityX != 1) {
                velocityX = -1;
                velocityY = 0;
            }
            break;
        case actions.right:
            if (velocityX != -1) {
                velocityX = 1;
                velocityY = 0;
            }
            break;
    }
};

const initGame = () => {
    if (gameOver) return handleGameOver();
    let htmlMarkup = `<div class="food" style="grid-area: ${foodY} / ${foodX}"></div>`;

    if (snakeX === foodX && snakeY === foodY) {
        changeFoodPosition();
        snakeBody.push([foodX, foodY]);
        score++;

        highScore = score >= highScore ? score : highScore;
        localStorage.setItem("high-score", highScore);
        scoreElement.innerText = `Score: ${score}`;
        highScoreElement.innerText = `High Score: ${highScore}`;
    }

    for (let i = snakeBody.length - 1; i > 0; i--) {
        snakeBody[i] = snakeBody[i - 1];
    }

    snakeBody[0] = [snakeX, snakeY];

    const reward = getReward();
    experiences.push({ state: getState(), reward });
    chooseAction();
    
    snakeX += velocityX;
    snakeY += velocityY;

    if (collisionWithBody() || snakeX <= 0 || snakeX > 30 || snakeY <= 0 || snakeY > 30) {
        gameOver = true;
    }

    for (let i = 0; i < snakeBody.length; i++) {
        htmlMarkup += `<div class="head" style="grid-area: ${snakeBody[i][1]} / ${snakeBody[i][0]}"></div>`;
    }

    playBoard.innerHTML = htmlMarkup;

    trainModel();
};

// Inicialização do jogo
changeFoodPosition();
setIntervalId = setInterval(initGame, 125);
