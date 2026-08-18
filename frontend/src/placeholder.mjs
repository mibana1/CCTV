import { createServer } from "node:http";

const host = "0.0.0.0";
const port = Number(process.env.PORT ?? 5173);

const page = `<!doctype html>
<html lang="ko">
  <head><meta charset="utf-8"><title>CCTV Test</title></head>
  <body>
    <main>
      <h1>CCTV Test</h1>
      <p>Frontend 기술 선택 전 임시 화면입니다.</p>
      <p>Backend health: <a href="http://127.0.0.1:8000/health">/health</a></p>
    </main>
  </body>
</html>`;

createServer((_request, response) => {
  response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  response.end(page);
}).listen(port, host, () => {
  console.log(`CCTV frontend placeholder listening on http://${host}:${port}`);
});

