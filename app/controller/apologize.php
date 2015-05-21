<?php

namespace app\controller;
use app\controller\base\controller;
use library\layout\components\alert;
use library\layout\elements\script;

class apologize extends controller{

    public function __construct($action,$uri,$error = false) {
        $this->error = $error;
        parent::__construct($action,$uri);
        
    }

    public function index(){

    }
    public function ajax(){
        $alert = new alert();
        $alert->create($this->error);
        script::addValue($alert,"'body'","prepend");
        script::getResponse();
    }
}