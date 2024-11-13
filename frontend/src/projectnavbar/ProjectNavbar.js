import React from "react";
import Container from "react-bootstrap/Container";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";

function ProjectNavbar() {
  return (
    <Navbar expand="lg" style={{ backgroundColor: "#03346E", position:'fixed',width:'100%'}}>
      <Container>
        {/* Navbar brand with text */}
        <Navbar.Brand
          className="text-white"
          style={{ fontFamily: "Lexend, sans-serif" }}
          href="/"
        >
          COVID Prediction
        </Navbar.Brand>
        {/* Toggle button for collapsed view */}
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          {/* Navigation Links */}
          <Nav className="ms-auto">
            <Nav.Link className="text-white" href="/upload">
              Predict
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default ProjectNavbar;
