export const metadata = {
  title: "ReAct QA Agent",
  description: "A simple web QA agent using ReAct reasoning",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}