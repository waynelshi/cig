export default function middleware(request) {
  const userAgent = request.headers.get('user-agent') || '';
  const url = new URL(request.url);

  // Intercept curl, wget, httpie at the root
  if (
    (url.pathname === '/' || url.pathname === '/index.html') &&
    /(curl|wget|httpie)/i.test(userAgent)
  ) {
    url.pathname = '/install.sh';
    return Response.rewrite(url);
  }
}

export const config = {
  matcher: ['/', '/index.html'],
};