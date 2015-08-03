<?php

namespace app\controller;
use app\controller\base\controller;
use library\layout\components\template;


class index extends controller{

    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }

    public function index(){
        $navbar = new template("templates/navbar","nav");
        $navbar->addClassName("navbar navbar-default navbar-fixed-top");
        $header = new template("templates/header");
        $index = new template("index/index");
        $footer = new template("templates/footer");
        $this->layout->addChild($navbar);
        $this->layout->addChild($header);
        $this->layout->addChild($index);
        $this->layout->addChild($footer);
        $this->html();
    }
    
    
    

}