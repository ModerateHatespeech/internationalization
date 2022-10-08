from sanic import Sanic
from sanic import response

from transformers import MarianMTModel, MarianTokenizer
import torch

from sanic.exceptions import SanicException, ServerError, NotFound
from sanic.response import text, json

import hashlib
import redis
import traceback

r = redis.StrictRedis(host="localhost", port=6379, db=0)

def cache(redis):
    """ Basic decorator that implements a simple Redis KV cache with an infinite TTL """

    def is_cached(key):
        """ Check if key (md5 of string) exists inside Redis """
        return redis.exists(key)

    def get_cached(key):
        """ Get key (md5 of string) from Redis """
        return redis.get(key)

    def prime_cache(key, value):
        """ If cached doesn't exist, insert intored cache """
        if not is_cached(key):
            redis.set(key, value)

    def decorator(fn):
        def wrapped(*args, **kwargs):
            """ Translate text into md5 hash as the key. Then, check if key exists from Redis, serving the cached value if it exists, otherwise running the translate call and saving the result """
            key = hashlib.md5(args[0].encode("utf-8")).hexdigest()

            cache = get_cached(key)
            if cache:
                return cache
            else:
                res = fn(*args, **kwargs)

            prime_cache(key, res)
            return res

        return wrapped

    return decorator

# initialize pretrained ML Model via HF and move to CUDA device if avaliable
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@cache(r)
def translate(text):
    """ Given (str) text, translate it into English """
    ouputs = model.generate(**tokenizer(text, return_tensors="pt"))
    return tokenizer.decode(ouputs[0], skip_special_tokens=True)

app = Sanic(__name__)

@app.route("/translate", methods=["POST"])
async def analysis(request):
    """
    Endpoint takes a POST request, with a JSON array passed as the body and the key "text" denoting the text to translate
    Returns a JSON encoded array with "result" as the translated string and "success" as True
    On failure, returns "success" as False and "result" with the error message
    Unexpected errors will be logged to docker logs for review
    """
    try:
        request_json = request.json
        if not "text" in request_json:
            return json({"result": "No text string passed", "success": False})
        return translate(request_json["text"])
    except Exception:
        print(traceback.format_exc())
        return json({"result": "Internal error", "success": False})

@app.exception(NotFound)
async def manage_not_found(request, exception):
    return text('{"response":"Invalid endpoint"}')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, workers=1)
