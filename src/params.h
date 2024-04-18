/*! \struct _learnParams_t
 *  \brief Hyper parameters for gradient descent
 *
 *  Contains all the hyper parameters for performing gradient descent. Uses a momentum based
 * approach with exponential decay of old gradient
 */
typedef struct _learnParams_t {
    float learningRate;  /*!< Gradient step size */
    float momentumDecay; /*!< Gradient decay for momentum */
    float regStrength;   /*!< Regularization strength for fully connected layers */
} learnParams_t;
