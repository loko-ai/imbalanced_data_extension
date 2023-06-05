import json

import uvicorn
from fastapi import Request, FastAPI, File
from loguru import logger
from loko_extensions.business.decorators import extract_value_args

from business.imblearn_methods import Resampler
from utils.decorator_fastapi import ExtractValueArgsFastapi

app = FastAPI(docs_url="/api")



@app.post("/balance")
@ExtractValueArgsFastapi(file=False)
def text_generator_api(value, args):
    logger.debug(f"args::: {args}")
    method = args.get("method", "undersampling")
    sampling_strategy = args.get("sampling_strategy", "auto")
    replacement = args.get("replacement", False)
    k_neighbors = int(args.get("k_neighbors", 5))
    random_state = int(args.get("random_state", 123))
    resampler = Resampler(method=method, random_state=random_state, sampling_strategy=sampling_strategy, replacement=replacement, k_neighbors=k_neighbors)
    data = eval(value)
    X = data.get("X")
    y = data.get("y")
    X_balanced, y_balanced = resampler(X, y)
    return dict(X=X_balanced, y=y_balanced)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

