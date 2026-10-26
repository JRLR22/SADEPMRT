import { render, screen } from '@testing-library/react';
import App from './App';

test('muestra la pantalla de inicio de sesión cuando no hay sesión', () => {
  localStorage.removeItem('token');
  render(<App />);
  expect(screen.getByText(/Iniciar sesión/i)).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/Correo institucional/i)).toBeInTheDocument();
});
