from fastapi import FastAPI


# 创建一个FastApi应用实例
app = FastAPI()

# 写一个测试接口
@app.get("/hello")
def hello():
    return {"msg":"helle FastAPI"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="test:app", host="127.0.0.1", port=8080)