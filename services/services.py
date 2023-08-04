import json

import uvicorn
from fastapi import Request, FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger
from loko_extensions.business.decorators import extract_value_args

from business.imblearn_methods import Resampler
from utils.data_utils import preprocess_dict_input
from utils.decorator_fastapi import ExtractValueArgsFastapi

app = FastAPI(docs_url="/api")


@app.exception_handler(Exception)
async def handle_exception(request, exc):
    # Log the exception
    logger.error(str(exc))

    # Propagate the exception as a JSON response
    response_data = {"detail": str(exc)}
    response_json = jsonable_encoder(response_data)
    return JSONResponse(status_code=500, content=response_json)


@app.post("/balance")
@ExtractValueArgsFastapi(file=False)
def text_generator_api(value, args):
    logger.debug(f"args::: {args}")
    target = args.get("target", "target")
    method = args.get("method", "undersampling")
    if len(method) == 0:
        method = "undersampling"
    sampling_strategy = args.get("sampling_strategy", "auto")
    if len(sampling_strategy) == 0:
        sampling_strategy = "auto"
    replacement = args.get("replacement", False)
    k_neighbors = args.get("k_neighbors", 5)
    if k_neighbors == "":
        k_neighbors = 5
    k_neighbors = int(k_neighbors)
    logger.debug(f"kkkkkkkkkkkkkk {k_neighbors}")
    random_state = int(args.get("random_state", 123))
    resampler = Resampler(method=method, random_state=random_state, sampling_strategy=sampling_strategy,
                          replacement=replacement, k_neighbors=k_neighbors)
    data = value
    X, y = preprocess_dict_input(data, target_variable=target)
    X_balanced, y_balanced = resampler(X, y)

    return dict(data=X_balanced.to_dict(orient="records"), target=y_balanced.to_list())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
