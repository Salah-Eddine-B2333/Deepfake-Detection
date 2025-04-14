import { useState } from "react";
import { Link } from "react-scroll";
import { FaTimes } from "react-icons/fa";
import { CiMenuFries } from "react-icons/ci";

const Nav = () => {
  const [click, setClick] = useState(false);
  const handleClick = () => {
    setClick(!click);
  };
  const content = (
    <>
      <div className="lg:hidden block absolute top-16 w-full left-0 right-0 bg-slate-900 transition">
        <ul className="text-center text-xl p-20 ">
          <Link spy={true} smooth={true} to="Home">
            <li className="my-4 py-4 border-b border-slate-800 hover:bg-slate-800 hover:rounded">
              Home
            </li>
          </Link>
          <Link spy={true} smooth={true} to="About">
            <li className="my-4 py-4 border-b border-slate-800 hover:bg-slate-800 hover:rounded">
              Services
            </li>
          </Link>
          <Link spy={true} smooth={true} to="Services">
            <li className="my-4 py-4 border-b border-slate-800 hover:bg-slate-800 hover:rounded">
              Contact
            </li>
          </Link>
          <a href="https://github.com/Salah-Eddine-B2333/Deepfake-Detection.git" target="_blank" rel="noopener noreferrer">
            <li className="transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
              Source-code
            </li>
          </a>

          <Link spy={true} smooth={true} to="Contact">
            <li className="my-4 py-4 border-b border-slate-800 hover:bg-slate-800 hover:rounded">
              About
            </li>
          </Link>
        </ul>
      </div>
    </>
  );

  return (
    <nav>
      <div className="h-10vh flex jsutify-between z-50 text-white lg:py-5 px-20 py-4 ">
        <div className="flex items-center flex-1">
          <span className="text-3xl font-bold">DF Detection</span>
        </div>
        <div className="lg:flex md:flex lg:flex-1 items-center justify-end font-normal hidden">
          <div className="flex-10 ">
            <ul className="flex gap-8 mr-16 text-[18px]">
              <Link spy={true} smooth={true} to="Home">
                <li className=" transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
                  Home
                </li>
              </Link>
              <Link spy={true} smooth={true} to="About">
                <li className=" transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
                  Services
                </li>
              </Link>
              <Link spy={true} smooth={true} to="Services">
                <li className=" transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
                  Contact
                </li>
              </Link>
              <a href="https://github.com/Salah-Eddine-B2333/Deepfake-Detection.git" target="_blank" rel="noopener noreferrer">
                <li className="transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
                  Source-code
                </li>
              </a>

              <Link spy={true} smooth={true} to="Contact">
                <li className=" transition border-b-2 border-slate-900 hover:border-blue-600 cursor-pointer">
                  About
                </li>
              </Link>
            </ul>
          </div>
        </div>
        <div>{click && content}</div>
        <button className="block sm:hidden transtion" onClick={handleClick}>
          {click ? <FaTimes /> : <CiMenuFries />}
        </button>
      </div>
    </nav>
  );
};

export default Nav;
