<?php

/**
 *  html class
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class html extends paired{
    public function __construct() {
        parent::__construct();
        $this->tag = "html";
    }
}
