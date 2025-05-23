import tensorflow as tf 
from keras.saving import register_keras_serializable

UNITS = 256
VOCAB_SIZE = 6000

eng_vectorizer_model_path = "app/model/eng_vectorizer_model.keras"
vie_vectorizer_model_path = "app/model/vie_vectorizer_model.keras"
vie_id2word_model_path = "app/model/vie_id2word_model.keras"
translator_model_path = "app/model/translator.keras"

@register_keras_serializable()
def clean(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^a-zA-ZÀ-ỹà-ỹ?.,! ]", "")
    text = tf.strings.regex_replace(text, "[?.,!]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
    return text

eng_vectorizer_model = tf.keras.models.load_model(eng_vectorizer_model_path)
eng_vectorizer = eng_vectorizer_model.layers[0]

vie_vectorizer_model = tf.keras.models.load_model(vie_vectorizer_model_path)
vie_vectorizer = vie_vectorizer_model.layers[0]


vie_id2word_model = tf.keras.models.load_model(vie_id2word_model_path)
vie_id2word = vie_id2word_model.layers[0]

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units = UNITS, vocab_size = VOCAB_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, 
            output_dim = units, 
            mask_zero = True
        )
        self.rnn = tf.keras.layers.LSTM(
            units = units,
            return_sequences = True, 
        )
        
    def call(self, context):
        """
        Forward pass of this layer

        Args:
            context (tf.Tensor) : The sentences to translate (B, L) 
            
        Returns:
            tf.Tensor: (B, L, D): Encoded sentence to tranlsate
        """
        x = self.embedding(context)
        x = self.rnn(x)
        return x

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim = units,
            num_heads = 1
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
    def call(self, context, target):
        """Forward pass of Attention

        Args:
            context (tf.Tensor): (B, L1, D) encoded setence to translate
            target (tf.Tensor): (B, L2, D) right shifted translation 
            
        Return:
            tf.Tensor: (B, L2, D) cross attention between context and target
        """
        # query is the translation and the value is the context 
        # default key = value 
        attn_output = self.mha(
            query = target,
            value = context
        )
        x = self.add([target, attn_output])
        x = self.layernorm(x)
        return x     

class Decoder(tf.keras.layers.Layer):
    def __init__(self, units = UNITS, vocab_size = VOCAB_SIZE):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = units, 
            mask_zero = True
        )
        self.pre_attn_rnn = tf.keras.layers.LSTM(
            units = units, 
            return_sequences = True,
            return_state = True 
        )
        self.attention = CrossAttention(units)
        self.post_attn_rnn = tf.keras.layers.LSTM(
            units = units, 
            return_sequences = True
        )
        self.dense = tf.keras.layers.Dense(
            units = vocab_size, 
            activation = "log_softmax"
        )
    def call(self,
             context,
             target,
             state = None,
             return_state = False):
        """Forwardpass of Decodeer

        Args:
            context (tf.Tensor): (B, L1, D) encoded sentence to translate 
            target (tf.Tensor): (B, L1) the right-shited translation 
            state (list([tf.Tensor, tf.Tensor]), optional): Hidden state of the pre-attn RNN. Defaults to None.
            return_state (bool, optional): Return state of pre-attn LSTM. Defaults to False.
        """
        x = self.embedding(target)
        
        x, h, c = self.pre_attn_rnn(
            x, 
            initial_state = state
        )
        
        x = self.attention(context, x)
        x = self.post_attn_rnn(x)
        logits = self.dense(x)
        
        if return_state: return logits, [h, c]
        return logits

class Translator(tf.keras.Model):
    def __init__(self, units=UNITS, vocab_size=VOCAB_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(units, vocab_size)
        self.decoder = Decoder(units, vocab_size)
    def call(self, inputs):
        """Forward pass of the model

        Args:
            inputs (tuple(tf.Tensor, tf.Tensor)): (2, B, L) Tuple of context and target 
        
        Returns:
            tf.Tensor: (B, L2, V) log_softmax probabilites of predict a particular token
        """
        context, target = inputs 
        encoded_context = self.encoder(context)
        logits = self.decoder(encoded_context, target)
        return logits
    
def masked_loss(y_true, y_pred):
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Check which elements of y_true are padding
    mask = tf.cast(y_true != 0, loss.dtype)
    
    loss *= mask
    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    match*= mask

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

def vect_to_text(vect, id2word):
    """
    Convert from vector of ids into the corresponding text

    Args:
        vect (tf.Tensor): include id of words in the sequence
        id2word (tf.keras.layers.StringLookup): invert string lookup for convert id to word

    Returns:
        str: the converted texts
    """
    words = id2word(vect)
    no_pad = tf.boolean_mask(words, words != b"")
    text = tf.strings.reduce_join(no_pad, separator=" ")
    return text.numpy().decode()


def generate_next_token(decoder, context, next_token, state, done, temperature=0.0):
    # Get the logits and state from the decoder
    logits, state = decoder(context, next_token, state=state, return_state=True)

    # Trim the intermediate dimension
    logits = logits[:, -1, :]

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # Trim dimensions of size 1
    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)

    # Get the logit of the selected next_token
    logit = logits[next_token].numpy()

    # Reshape to (1,1) since this is the expected shape for text encoded as TF tensors
    next_token = tf.reshape(next_token, shape=(1, 1))

    # If next_token is End-of-Sentence token you are done
    if next_token == 3:  # [EOS]
        done = True

    return next_token, logit, state, done


def translate(model, text, max_length=15, temperature=0.0):
    """Translate a English sentence to Vietnamese

    Args:
        model (tf.keras.Model): The translator model
        text (str): The English context sentence
        max_length (int, optional): The maximum length of the translation. Defaults to 15.
        temperature (float, optional): Randomness of the translation. Defaults to 0.0.

    Returns:
        tuple(str, np.float, tf.Tensor): The translation, logit that predicted [EOS] token, and the tokenizeed translation
    """
    tokens, logits = list(), list()

    # convert the string to tensor
    text = tf.convert_to_tensor(text)[tf.newaxis]

    # vectorize the string tensor
    context = eng_vectorizer(text).to_tensor()

    # get the encoded context
    context = model.encoder(context)

    # [SOS]
    next_token = tf.fill((1, 1), 2)

    # initial hidden states(h0) and cell states(c0) will be tensor of 0 with dim (1, UNITS)
    state = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]

    # done flag
    done = False
    for i in range(max_length):
        try:
            next_token, logit, state, done = generate_next_token(
                decoder=model.decoder,
                context=context,
                next_token=next_token,
                state=state,
                done=done,
                temperature=temperature,
            )
        except:
            raise Exception("Problem generating the next token")

        # check done
        if done:
            break

        # append the result
        tokens.append(next_token)
        logits.append(logit)

    # concatenate all token into a tensor
    tokens = tf.concat(tokens, axis=-1)

    # convert translated tokens to text
    translation = vect_to_text(tokens, vie_id2word)

    return translation, logits[-1], tokens




loaded_translator = tf.keras.models.load_model(
    translator_model_path,
    custom_objects={
        "Translator": Translator,
        "Encoder": Encoder,
        "Decoder": Decoder,
        "CrossAttention": CrossAttention,
        "masked_loss": masked_loss,
        "masked_acc": masked_acc,
    },
)
