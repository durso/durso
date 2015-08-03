<?php

namespace app\controller;
use app\controller\base\controller;


class apologize extends controller{

    public function __construct($action,$uri,$error = false) {
        $this->error = $error;
        parent::__construct($action,$uri);
        
    }

    public function index(){
        die($this->error);
    }
    public function ajax(){
        header('HTTP/1.1 500 Internal Server Error');
        die($this->error);
    }
}