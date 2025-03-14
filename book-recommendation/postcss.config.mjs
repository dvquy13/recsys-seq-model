const config = {
  plugins: [
    require('@tailwindcss/postcss'),
    {
      theme: {
        extend: {
          fontFamily: {
            inter: ['InterVariable', 'Inter', 'sans-serif'],
          },
        },
      },
    },
  ],
};

export default config;
