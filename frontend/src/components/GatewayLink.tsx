// Floating top-left link back to the gateway launcher ("The Web").
// Copied verbatim across apps (house convention — no shared packages).
// The gateway serves the apex of whatever domain served this app, so the
// URL is derived by stripping the first host label; on localhost/raw-IP
// dev there is no gateway to return to, so the button renders nothing.
const IPV4 = /^\d{1,3}(\.\d{1,3}){3}$/

function gatewayUrl(): string | null {
  const { hostname, port, protocol } = window.location
  const labels = hostname.split('.')
  if (labels.length <= 1 || IPV4.test(hostname)) return null
  return `${protocol}//${labels.slice(1).join('.')}${port ? `:${port}` : ''}/`
}

export function GatewayLink() {
  const url = gatewayUrl()
  if (!url) return null
  return (
    <a className="gateway-link" href={url} aria-label="Back to The Web" title="The Web">
      <img src="/web.png" alt="" />
    </a>
  )
}
